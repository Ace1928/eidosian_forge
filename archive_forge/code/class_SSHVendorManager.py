import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
class SSHVendorManager:
    """Manager for manage SSH vendors."""

    def __init__(self):
        self._ssh_vendors = {}
        self._cached_ssh_vendor = None
        self._default_ssh_vendor = None

    def register_default_vendor(self, vendor):
        """Register default SSH vendor."""
        self._default_ssh_vendor = vendor

    def register_vendor(self, name, vendor):
        """Register new SSH vendor by name."""
        self._ssh_vendors[name] = vendor

    def clear_cache(self):
        """Clear previously cached lookup result."""
        self._cached_ssh_vendor = None

    def _get_vendor_by_config(self):
        vendor_name = config.GlobalStack().get('ssh')
        if vendor_name is not None:
            try:
                vendor = self._ssh_vendors[vendor_name]
            except KeyError:
                vendor = self._get_vendor_from_path(vendor_name)
                if vendor is None:
                    raise errors.UnknownSSH(vendor_name)
                vendor.executable_path = vendor_name
            return vendor
        return None

    def _get_ssh_version_string(self, args):
        """Return SSH version string from the subprocess."""
        try:
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, **os_specific_subprocess_params())
            stdout, stderr = p.communicate()
        except OSError:
            stdout = stderr = b''
        return (stdout + stderr).decode(osutils.get_terminal_encoding())

    def _get_vendor_by_version_string(self, version, progname):
        """Return the vendor or None based on output from the subprocess.

        :param version: The output of 'ssh -V' like command.
        :param args: Command line that was run.
        """
        vendor = None
        if 'OpenSSH' in version:
            trace.mutter('ssh implementation is OpenSSH')
            vendor = OpenSSHSubprocessVendor()
        elif 'SSH Secure Shell' in version:
            trace.mutter('ssh implementation is SSH Corp.')
            vendor = SSHCorpSubprocessVendor()
        elif 'lsh' in version:
            trace.mutter('ssh implementation is GNU lsh.')
            vendor = LSHSubprocessVendor()
        elif 'plink' in version and progname == 'plink':
            trace.mutter("ssh implementation is Putty's plink.")
            vendor = PLinkSubprocessVendor()
        return vendor

    def _get_vendor_by_inspection(self):
        """Return the vendor or None by checking for known SSH implementations."""
        version = self._get_ssh_version_string(['ssh', '-V'])
        return self._get_vendor_by_version_string(version, 'ssh')

    def _get_vendor_from_path(self, path):
        """Return the vendor or None using the program at the given path"""
        version = self._get_ssh_version_string([path, '-V'])
        return self._get_vendor_by_version_string(version, os.path.splitext(os.path.basename(path))[0])

    def get_vendor(self):
        """Find out what version of SSH is on the system.

        :raises SSHVendorNotFound: if no any SSH vendor is found
        :raises UnknownSSH: if the BRZ_SSH environment variable contains
                            unknown vendor name
        """
        if self._cached_ssh_vendor is None:
            vendor = self._get_vendor_by_config()
            if vendor is None:
                vendor = self._get_vendor_by_inspection()
                if vendor is None:
                    trace.mutter('falling back to default implementation')
                    vendor = self._default_ssh_vendor
                    if vendor is None:
                        raise errors.SSHVendorNotFound()
            self._cached_ssh_vendor = vendor
        return self._cached_ssh_vendor