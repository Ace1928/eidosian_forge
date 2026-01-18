import subprocess

from . import SystemInterface

class FedoraInterface(SystemInterface):
    """
        SystemInterface implementation for Fedora hosts. Contains commands
        that are specific to the Fedora toolchain.
    """

    def get_arch(self):
        return ['uname', '-m']

    def get_installed_packages(self, target_file):
        return ['sh', '-ec', 'rpm -qa > %s' % target_file.tb]

    def can_query_packages(self):
        try:
            return subprocess.check_call(['which', 'rpm'],stdout=subprocess.DEVNULL) == 0
        except subprocess.CalledProcessError:
            return 0
