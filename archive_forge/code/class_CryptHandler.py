from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
class CryptHandler(Handler):

    def __init__(self, module):
        super(CryptHandler, self).__init__(module)
        self._cryptsetup_bin = self._module.get_bin_path('cryptsetup', True)

    def get_container_name_by_device(self, device):
        """ obtain LUKS container name based on the device where it is located
            return None if not found
            raise ValueError if lsblk command fails
        """
        result = self._run_command([self._lsblk_bin, device, '-nlo', 'type,name'])
        if result[RETURN_CODE] != 0:
            raise ValueError('Error while obtaining LUKS name for %s: %s' % (device, result[STDERR]))
        for line in result[STDOUT].splitlines(False):
            m = LUKS_NAME_REGEX.match(line)
            if m:
                return m.group(1)
        return None

    def get_container_device_by_name(self, name):
        """ obtain device name based on the LUKS container name
            return None if not found
            raise ValueError if lsblk command fails
        """
        result = self._run_command([self._cryptsetup_bin, 'status', name])
        if result[RETURN_CODE] != 0:
            return None
        m = LUKS_DEVICE_REGEX.search(result[STDOUT])
        device = m.group(1)
        return device

    def is_luks(self, device):
        """ check if the LUKS container does exist
        """
        result = self._run_command([self._cryptsetup_bin, 'isLuks', device])
        return result[RETURN_CODE] == 0

    def get_luks_type(self, device):
        """ get the luks type of a device
        """
        if self.is_luks(device):
            with open(device, 'rb') as f:
                for offset in LUKS2_HEADER_OFFSETS:
                    f.seek(offset)
                    data = f.read(LUKS_HEADER_L)
                    if data == LUKS2_HEADER2:
                        return 'luks2'
                return 'luks1'
        return None

    def is_luks_slot_set(self, device, keyslot):
        """ check if a keyslot is set
        """
        result = self._run_command([self._cryptsetup_bin, 'luksDump', device])
        if result[RETURN_CODE] != 0:
            raise ValueError('Error while dumping LUKS header from %s' % (device,))
        result_luks1 = 'Key Slot %d: ENABLED' % keyslot in result[STDOUT]
        result_luks2 = ' %d: luks2' % keyslot in result[STDOUT]
        return result_luks1 or result_luks2

    def _add_pbkdf_options(self, options, pbkdf):
        if pbkdf['iteration_time'] is not None:
            options.extend(['--iter-time', str(int(pbkdf['iteration_time'] * 1000))])
        if pbkdf['iteration_count'] is not None:
            options.extend(['--pbkdf-force-iterations', str(pbkdf['iteration_count'])])
        if pbkdf['algorithm'] is not None:
            options.extend(['--pbkdf', pbkdf['algorithm']])
        if pbkdf['memory'] is not None:
            options.extend(['--pbkdf-memory', str(pbkdf['memory'])])
        if pbkdf['parallel'] is not None:
            options.extend(['--pbkdf-parallel', str(pbkdf['parallel'])])

    def run_luks_create(self, device, keyfile, passphrase, keyslot, keysize, cipher, hash_, sector_size, pbkdf):
        luks_type = self._module.params['type']
        label = self._module.params['label']
        options = []
        if keysize is not None:
            options.append('--key-size=' + str(keysize))
        if label is not None:
            options.extend(['--label', label])
            luks_type = 'luks2'
        if luks_type is not None:
            options.extend(['--type', luks_type])
        if cipher is not None:
            options.extend(['--cipher', cipher])
        if hash_ is not None:
            options.extend(['--hash', hash_])
        if pbkdf is not None:
            self._add_pbkdf_options(options, pbkdf)
        if sector_size is not None:
            options.extend(['--sector-size', str(sector_size)])
        if keyslot is not None:
            options.extend(['--key-slot', str(keyslot)])
        args = [self._cryptsetup_bin, 'luksFormat']
        args.extend(options)
        args.extend(['-q', device])
        if keyfile:
            args.append(keyfile)
        result = self._run_command(args, data=passphrase)
        if result[RETURN_CODE] != 0:
            raise ValueError('Error while creating LUKS on %s: %s' % (device, result[STDERR]))

    def run_luks_open(self, device, keyfile, passphrase, perf_same_cpu_crypt, perf_submit_from_crypt_cpus, perf_no_read_workqueue, perf_no_write_workqueue, persistent, allow_discards, name):
        args = [self._cryptsetup_bin]
        if keyfile:
            args.extend(['--key-file', keyfile])
        if perf_same_cpu_crypt:
            args.extend(['--perf-same_cpu_crypt'])
        if perf_submit_from_crypt_cpus:
            args.extend(['--perf-submit_from_crypt_cpus'])
        if perf_no_read_workqueue:
            args.extend(['--perf-no_read_workqueue'])
        if perf_no_write_workqueue:
            args.extend(['--perf-no_write_workqueue'])
        if persistent:
            args.extend(['--persistent'])
        if allow_discards:
            args.extend(['--allow-discards'])
        args.extend(['open', '--type', 'luks', device, name])
        result = self._run_command(args, data=passphrase)
        if result[RETURN_CODE] != 0:
            raise ValueError('Error while opening LUKS container on %s: %s' % (device, result[STDERR]))

    def run_luks_close(self, name):
        result = self._run_command([self._cryptsetup_bin, 'close', name])
        if result[RETURN_CODE] != 0:
            raise ValueError('Error while closing LUKS container %s' % name)

    def run_luks_remove(self, device):
        wipefs_bin = self._module.get_bin_path('wipefs', True)
        name = self.get_container_name_by_device(device)
        if name is not None:
            self.run_luks_close(name)
        result = self._run_command([wipefs_bin, '--all', device])
        if result[RETURN_CODE] != 0:
            raise ValueError('Error while wiping LUKS container signatures for %s: %s' % (device, result[STDERR]))
        try:
            wipe_luks_headers(device)
        except Exception as exc:
            raise ValueError('Error while wiping LUKS container signatures for %s: %s' % (device, exc))

    def run_luks_add_key(self, device, keyfile, passphrase, new_keyfile, new_passphrase, new_keyslot, pbkdf):
        """ Add new key from a keyfile or passphrase to given 'device';
            authentication done using 'keyfile' or 'passphrase'.
            Raises ValueError when command fails.
        """
        data = []
        args = [self._cryptsetup_bin, 'luksAddKey', device]
        if pbkdf is not None:
            self._add_pbkdf_options(args, pbkdf)
        if new_keyslot is not None:
            args.extend(['--key-slot', str(new_keyslot)])
        if keyfile:
            args.extend(['--key-file', keyfile])
        else:
            data.append(passphrase)
        if new_keyfile:
            args.append(new_keyfile)
        else:
            data.extend([new_passphrase, new_passphrase])
        result = self._run_command(args, data='\n'.join(data) or None)
        if result[RETURN_CODE] != 0:
            raise ValueError('Error while adding new LUKS keyslot to %s: %s' % (device, result[STDERR]))

    def run_luks_remove_key(self, device, keyfile, passphrase, keyslot, force_remove_last_key=False):
        """ Remove key from given device
            Raises ValueError when command fails
        """
        if not force_remove_last_key:
            result = self._run_command([self._cryptsetup_bin, 'luksDump', device])
            if result[RETURN_CODE] != 0:
                raise ValueError('Error while dumping LUKS header from %s' % (device,))
            keyslot_count = 0
            keyslot_area = False
            keyslot_re = re.compile('^Key Slot [0-9]+: ENABLED')
            for line in result[STDOUT].splitlines():
                if line.startswith('Keyslots:'):
                    keyslot_area = True
                elif line.startswith('  '):
                    if keyslot_area and line[2] in '0123456789':
                        keyslot_count += 1
                elif line.startswith('\t'):
                    pass
                elif keyslot_re.match(line):
                    keyslot_count += 1
                else:
                    keyslot_area = False
            if keyslot_count < 2:
                self._module.fail_json(msg='LUKS device %s has less than two active keyslots. To be able to remove a key, please set `force_remove_last_key` to `true`.' % device)
        if keyslot is None:
            args = [self._cryptsetup_bin, 'luksRemoveKey', device, '-q']
        else:
            args = [self._cryptsetup_bin, 'luksKillSlot', device, '-q', str(keyslot)]
        if keyfile:
            args.extend(['--key-file', keyfile])
        result = self._run_command(args, data=passphrase)
        if result[RETURN_CODE] != 0:
            raise ValueError('Error while removing LUKS key from %s: %s' % (device, result[STDERR]))

    def luks_test_key(self, device, keyfile, passphrase, keyslot=None):
        """ Check whether the keyfile or passphrase works.
            Raises ValueError when command fails.
        """
        data = None
        args = [self._cryptsetup_bin, 'luksOpen', '--test-passphrase', device]
        if keyfile:
            args.extend(['--key-file', keyfile])
        else:
            data = passphrase
        if keyslot is not None:
            args.extend(['--key-slot', str(keyslot)])
        result = self._run_command(args, data=data)
        if result[RETURN_CODE] == 0:
            return True
        for output in (STDOUT, STDERR):
            if 'No key available with this passphrase' in result[output]:
                return False
            if 'No usable keyslot is available.' in result[output]:
                return False
        if result[RETURN_CODE] == 1 and keyslot is not None and (result[STDOUT] == '') and (result[STDERR] == ''):
            return False
        raise ValueError('Error while testing whether keyslot exists on %s: %s' % (device, result[STDERR]))