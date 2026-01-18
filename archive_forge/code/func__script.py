import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def _script():
    help = "    %s [--user-base] [--user-site]\n\n    Without arguments print some useful information\n    With arguments print the value of USER_BASE and/or USER_SITE separated\n    by '%s'.\n\n    Exit codes with --user-base or --user-site:\n      0 - user site directory is enabled\n      1 - user site directory is disabled by user\n      2 - user site directory is disabled by super user\n          or for security reasons\n     >2 - unknown error\n    "
    args = sys.argv[1:]
    if not args:
        user_base = getuserbase()
        user_site = getusersitepackages()
        print('sys.path = [')
        for dir in sys.path:
            print('    %r,' % (dir,))
        print(']')

        def exists(path):
            if path is not None and os.path.isdir(path):
                return 'exists'
            else:
                return "doesn't exist"
        print(f'USER_BASE: {user_base!r} ({exists(user_base)})')
        print(f'USER_SITE: {user_site!r} ({exists(user_site)})')
        print(f'ENABLE_USER_SITE: {ENABLE_USER_SITE!r}')
        sys.exit(0)
    buffer = []
    if '--user-base' in args:
        buffer.append(USER_BASE)
    if '--user-site' in args:
        buffer.append(USER_SITE)
    if buffer:
        print(os.pathsep.join(buffer))
        if ENABLE_USER_SITE:
            sys.exit(0)
        elif ENABLE_USER_SITE is False:
            sys.exit(1)
        elif ENABLE_USER_SITE is None:
            sys.exit(2)
        else:
            sys.exit(3)
    else:
        import textwrap
        print(textwrap.dedent(help % (sys.argv[0], os.pathsep)))
        sys.exit(10)