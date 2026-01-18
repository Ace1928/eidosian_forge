from __future__ import (absolute_import, division, print_function)
import errno
import fcntl
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings
from binascii import hexlify
from binascii import unhexlify
from binascii import Error as BinasciiError
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible import constants as C
from ansible.module_utils.six import binary_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe, unfrackpath
class VaultEditor:

    def __init__(self, vault=None):
        self.vault = vault or VaultLib()

    def _shred_file_custom(self, tmp_path):
        """"Destroy a file, when shred (core-utils) is not available

        Unix `shred' destroys files "so that they can be recovered only with great difficulty with
        specialised hardware, if at all". It is based on the method from the paper
        "Secure Deletion of Data from Magnetic and Solid-State Memory",
        Proceedings of the Sixth USENIX Security Symposium (San Jose, California, July 22-25, 1996).

        We do not go to that length to re-implement shred in Python; instead, overwriting with a block
        of random data should suffice.

        See https://github.com/ansible/ansible/pull/13700 .
        """
        file_len = os.path.getsize(tmp_path)
        if file_len > 0:
            max_chunk_len = min(1024 * 1024 * 2, file_len)
            passes = 3
            with open(tmp_path, 'wb') as fh:
                for dummy in range(passes):
                    fh.seek(0, 0)
                    chunk_len = random.randint(max_chunk_len // 2, max_chunk_len)
                    data = os.urandom(chunk_len)
                    for dummy in range(0, file_len // chunk_len):
                        fh.write(data)
                    fh.write(data[:file_len % chunk_len])
                    if fh.tell() != file_len:
                        raise AnsibleAssertionError()
                    os.fsync(fh)

    def _shred_file(self, tmp_path):
        """Securely destroy a decrypted file

        Note standard limitations of GNU shred apply (For flash, overwriting would have no effect
        due to wear leveling; for other storage systems, the async kernel->filesystem->disk calls never
        guarantee data hits the disk; etc). Furthermore, if your tmp dirs is on tmpfs (ramdisks),
        it is a non-issue.

        Nevertheless, some form of overwriting the data (instead of just removing the fs index entry) is
        a good idea. If shred is not available (e.g. on windows, or no core-utils installed), fall back on
        a custom shredding method.
        """
        if not os.path.isfile(tmp_path):
            return
        try:
            r = subprocess.call(['shred', tmp_path])
        except (OSError, ValueError):
            r = 1
        if r != 0:
            self._shred_file_custom(tmp_path)
        os.remove(tmp_path)

    def _edit_file_helper(self, filename, secret, existing_data=None, force_save=False, vault_id=None):
        root, ext = os.path.splitext(os.path.realpath(filename))
        fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=C.DEFAULT_LOCAL_TMP)
        cmd = self._editor_shell_command(tmp_path)
        try:
            if existing_data:
                self.write_data(existing_data, fd, shred=False)
        except Exception:
            self._shred_file(tmp_path)
            raise
        finally:
            os.close(fd)
        try:
            subprocess.call(cmd)
        except Exception as e:
            self._shred_file(tmp_path)
            raise AnsibleError('Unable to execute the command "%s": %s' % (' '.join(cmd), to_native(e)))
        b_tmpdata = self.read_data(tmp_path)
        if force_save or existing_data != b_tmpdata:
            b_ciphertext = self.vault.encrypt(b_tmpdata, secret, vault_id=vault_id)
            self.write_data(b_ciphertext, tmp_path)
            self.shuffle_files(tmp_path, filename)
            display.vvvvv(u'Saved edited file "%s" encrypted using %s and  vault id "%s"' % (to_text(filename), to_text(secret), to_text(vault_id)))
        self._shred_file(tmp_path)

    def _real_path(self, filename):
        if filename == '-':
            return filename
        real_path = os.path.realpath(filename)
        return real_path

    def encrypt_bytes(self, b_plaintext, secret, vault_id=None):
        b_ciphertext = self.vault.encrypt(b_plaintext, secret, vault_id=vault_id)
        return b_ciphertext

    def encrypt_file(self, filename, secret, vault_id=None, output_file=None):
        filename = self._real_path(filename)
        b_plaintext = self.read_data(filename)
        b_ciphertext = self.vault.encrypt(b_plaintext, secret, vault_id=vault_id)
        self.write_data(b_ciphertext, output_file or filename)

    def decrypt_file(self, filename, output_file=None):
        filename = self._real_path(filename)
        ciphertext = self.read_data(filename)
        try:
            plaintext = self.vault.decrypt(ciphertext, filename=filename)
        except AnsibleError as e:
            raise AnsibleError('%s for %s' % (to_native(e), to_native(filename)))
        self.write_data(plaintext, output_file or filename, shred=False)

    def create_file(self, filename, secret, vault_id=None):
        """ create a new encrypted file """
        dirname = os.path.dirname(filename)
        if dirname and (not os.path.exists(dirname)):
            display.warning(u'%s does not exist, creating...' % to_text(dirname))
            makedirs_safe(dirname)
        if os.path.isfile(filename):
            raise AnsibleError("%s exists, please use 'edit' instead" % filename)
        self._edit_file_helper(filename, secret, vault_id=vault_id)

    def edit_file(self, filename):
        vault_id_used = None
        vault_secret_used = None
        filename = self._real_path(filename)
        b_vaulttext = self.read_data(filename)
        vaulttext = to_text(b_vaulttext)
        try:
            plaintext, vault_id_used, vault_secret_used = self.vault.decrypt_and_get_vault_id(vaulttext)
        except AnsibleError as e:
            raise AnsibleError('%s for %s' % (to_native(e), to_native(filename)))
        dummy, dummy, cipher_name, vault_id = parse_vaulttext_envelope(b_vaulttext, filename=filename)
        force_save = cipher_name not in CIPHER_WRITE_WHITELIST
        self._edit_file_helper(filename, vault_secret_used, existing_data=plaintext, force_save=force_save, vault_id=vault_id)

    def plaintext(self, filename):
        b_vaulttext = self.read_data(filename)
        vaulttext = to_text(b_vaulttext)
        try:
            plaintext = self.vault.decrypt(vaulttext, filename=filename)
            return plaintext
        except AnsibleError as e:
            raise AnsibleVaultError('%s for %s' % (to_native(e), to_native(filename)))

    def rekey_file(self, filename, new_vault_secret, new_vault_id=None):
        filename = self._real_path(filename)
        prev = os.stat(filename)
        b_vaulttext = self.read_data(filename)
        vaulttext = to_text(b_vaulttext)
        display.vvvvv(u'Rekeying file "%s" to with new vault-id "%s" and vault secret %s' % (to_text(filename), to_text(new_vault_id), to_text(new_vault_secret)))
        try:
            plaintext, vault_id_used, _dummy = self.vault.decrypt_and_get_vault_id(vaulttext)
        except AnsibleError as e:
            raise AnsibleError('%s for %s' % (to_native(e), to_native(filename)))
        if new_vault_secret is None:
            raise AnsibleError('The value for the new_password to rekey %s with is not valid' % filename)
        new_vault = VaultLib(secrets={})
        b_new_vaulttext = new_vault.encrypt(plaintext, new_vault_secret, vault_id=new_vault_id)
        self.write_data(b_new_vaulttext, filename)
        os.chmod(filename, prev.st_mode)
        os.chown(filename, prev.st_uid, prev.st_gid)
        display.vvvvv(u'Rekeyed file "%s" (decrypted with vault id "%s") was encrypted with new vault-id "%s" and vault secret %s' % (to_text(filename), to_text(vault_id_used), to_text(new_vault_id), to_text(new_vault_secret)))

    def read_data(self, filename):
        try:
            if filename == '-':
                data = sys.stdin.buffer.read()
            else:
                with open(filename, 'rb') as fh:
                    data = fh.read()
        except Exception as e:
            msg = to_native(e)
            if not msg:
                msg = repr(e)
            raise AnsibleError('Unable to read source file (%s): %s' % (to_native(filename), msg))
        return data

    def write_data(self, data, thefile, shred=True, mode=384):
        """Write the data bytes to given path

        This is used to write a byte string to a file or stdout. It is used for
        writing the results of vault encryption or decryption. It is used for
        saving the ciphertext after encryption and it is also used for saving the
        plaintext after decrypting a vault. The type of the 'data' arg should be bytes,
        since in the plaintext case, the original contents can be of any text encoding
        or arbitrary binary data.

        When used to write the result of vault encryption, the value of the 'data' arg
        should be a utf-8 encoded byte string and not a text type.

        When used to write the result of vault decryption, the value of the 'data' arg
        should be a byte string and not a text type.

        :arg data: the byte string (bytes) data
        :arg thefile: file descriptor or filename to save 'data' to.
        :arg shred: if shred==True, make sure that the original data is first shredded so that is cannot be recovered.
        :returns: None
        """
        b_file_data = to_bytes(data, errors='strict')
        is_fd = False
        try:
            is_fd = isinstance(thefile, int) and fcntl.fcntl(thefile, fcntl.F_GETFD) != -1
        except Exception:
            pass
        if is_fd:
            os.ftruncate(thefile, 0)
            os.write(thefile, b_file_data)
        elif thefile == '-':
            output = getattr(sys.stdout, 'buffer', sys.stdout)
            output.write(b_file_data)
        else:
            if not os.access(os.path.dirname(thefile), os.W_OK):
                raise AnsibleError("Destination '%s' not writable" % os.path.dirname(thefile))
            if os.path.isfile(thefile):
                if shred:
                    self._shred_file(thefile)
                else:
                    os.remove(thefile)
            current_umask = os.umask(63)
            try:
                try:
                    fd = os.open(thefile, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_TRUNC, mode)
                except OSError as ose:
                    if ose.errno == errno.EEXIST:
                        raise AnsibleError('Vault file got recreated while we were operating on it: %s' % to_native(ose))
                    raise AnsibleError('Problem creating temporary vault file: %s' % to_native(ose))
                try:
                    os.ftruncate(fd, 0)
                    os.write(fd, b_file_data)
                except OSError as e:
                    raise AnsibleError('Unable to write to temporary vault file: %s' % to_native(e))
                finally:
                    os.close(fd)
            finally:
                os.umask(current_umask)

    def shuffle_files(self, src, dest):
        prev = None
        if os.path.isfile(dest):
            prev = os.stat(dest)
            os.remove(dest)
        shutil.move(src, dest)
        if prev is not None:
            os.chmod(dest, prev.st_mode)
            os.chown(dest, prev.st_uid, prev.st_gid)

    def _editor_shell_command(self, filename):
        env_editor = C.config.get_config_value('EDITOR')
        editor = shlex.split(env_editor)
        editor.append(filename)
        return editor