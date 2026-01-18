from boto.pyami.installers.ubuntu.installer import Installer
class Apache(Installer):
    """
    Install apache2, mod_python, and libapache2-svn
    """

    def install(self):
        self.run('apt-get update')
        self.run('apt-get -y install apache2', notify=True, exit_on_error=True)
        self.run('apt-get -y install libapache2-mod-python', notify=True, exit_on_error=True)
        self.run('a2enmod rewrite', notify=True, exit_on_error=True)
        self.run('a2enmod ssl', notify=True, exit_on_error=True)
        self.run('a2enmod proxy', notify=True, exit_on_error=True)
        self.run('a2enmod proxy_ajp', notify=True, exit_on_error=True)
        self.stop('apache2')
        self.start('apache2')

    def main(self):
        self.install()