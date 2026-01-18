from typing import List, Optional
class _UserRecord:
    """
    L{_UserRecord} holds the user data for a single user in L{UserDatabase}.
    It corresponds to the C{passwd} structure from the L{pwd} module.
    See that module for attribute documentation.
    """

    def __init__(self, name: str, password: str, uid: int, gid: int, gecos: str, home: str, shell: str) -> None:
        self.pw_name = name
        self.pw_passwd = password
        self.pw_uid = uid
        self.pw_gid = gid
        self.pw_gecos = gecos
        self.pw_dir = home
        self.pw_shell = shell

    def __len__(self) -> int:
        return 7

    def __getitem__(self, index):
        return (self.pw_name, self.pw_passwd, self.pw_uid, self.pw_gid, self.pw_gecos, self.pw_dir, self.pw_shell)[index]