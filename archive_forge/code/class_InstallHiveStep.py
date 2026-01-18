from boto.compat import six
class InstallHiveStep(HiveBase):
    """
    Install Hive on EMR step
    """
    InstallHiveName = 'Install Hive'

    def __init__(self, hive_versions='latest', hive_site=None):
        step_args = []
        step_args.extend(self.BaseArgs)
        step_args.extend(['--install-hive'])
        step_args.extend(['--hive-versions', hive_versions])
        if hive_site is not None:
            step_args.extend(['--hive-site=%s' % hive_site])
        super(InstallHiveStep, self).__init__(self.InstallHiveName, step_args=step_args)