import os, shlex, stat
def _security_check(self, fp, default_netrc, login):
    if os.name == 'posix' and default_netrc and (login != 'anonymous'):
        prop = os.fstat(fp.fileno())
        if prop.st_uid != os.getuid():
            import pwd
            try:
                fowner = pwd.getpwuid(prop.st_uid)[0]
            except KeyError:
                fowner = 'uid %s' % prop.st_uid
            try:
                user = pwd.getpwuid(os.getuid())[0]
            except KeyError:
                user = 'uid %s' % os.getuid()
            raise NetrcParseError(f'~/.netrc file owner ({fowner}, {user}) does not match current user')
        if prop.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise NetrcParseError('~/.netrc access too permissive: access permissions must restrict access to only the owner')