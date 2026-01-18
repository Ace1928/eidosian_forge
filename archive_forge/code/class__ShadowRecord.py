from typing import List, Optional
class _ShadowRecord:
    """
    L{_ShadowRecord} holds the shadow user data for a single user in
    L{ShadowDatabase}.  It corresponds to C{spwd.struct_spwd}.  See that class
    for attribute documentation.
    """

    def __init__(self, username: str, password: str, lastChange: int, min: int, max: int, warn: int, inact: int, expire: int, flag: int) -> None:
        self.sp_nam = username
        self.sp_pwd = password
        self.sp_lstchg = lastChange
        self.sp_min = min
        self.sp_max = max
        self.sp_warn = warn
        self.sp_inact = inact
        self.sp_expire = expire
        self.sp_flag = flag

    def __len__(self) -> int:
        return 9

    def __getitem__(self, index):
        return (self.sp_nam, self.sp_pwd, self.sp_lstchg, self.sp_min, self.sp_max, self.sp_warn, self.sp_inact, self.sp_expire, self.sp_flag)[index]