from __future__ import annotations
import errno
import os
import site
import stat
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
import platformdirs
from .utils import deprecation
def _win32_restrict_file_to_user_ctypes(fname: str) -> None:
    """Secure a windows file to read-only access for the user.

    Follows guidance from win32 library creator:
    http://timgolden.me.uk/python/win32_how_do_i/add-security-to-a-file.html

    This method should be executed against an already generated file which
    has no secrets written to it yet.

    Parameters
    ----------

    fname : unicode
        The path to the file to secure
    """
    import ctypes
    from ctypes import wintypes
    advapi32 = ctypes.WinDLL('advapi32', use_last_error=True)
    secur32 = ctypes.WinDLL('secur32', use_last_error=True)
    NameSamCompatible = 2
    WinBuiltinAdministratorsSid = 26
    DACL_SECURITY_INFORMATION = 4
    ACL_REVISION = 2
    ERROR_INSUFFICIENT_BUFFER = 122
    ERROR_MORE_DATA = 234
    SYNCHRONIZE = 1048576
    DELETE = 65536
    STANDARD_RIGHTS_REQUIRED = 983040
    STANDARD_RIGHTS_READ = 131072
    STANDARD_RIGHTS_WRITE = 131072
    FILE_READ_DATA = 1
    FILE_READ_EA = 8
    FILE_READ_ATTRIBUTES = 128
    FILE_WRITE_DATA = 2
    FILE_APPEND_DATA = 4
    FILE_WRITE_EA = 16
    FILE_WRITE_ATTRIBUTES = 256
    FILE_ALL_ACCESS = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 511
    FILE_GENERIC_READ = STANDARD_RIGHTS_READ | FILE_READ_DATA | FILE_READ_ATTRIBUTES | FILE_READ_EA | SYNCHRONIZE
    FILE_GENERIC_WRITE = STANDARD_RIGHTS_WRITE | FILE_WRITE_DATA | FILE_WRITE_ATTRIBUTES | FILE_WRITE_EA | FILE_APPEND_DATA | SYNCHRONIZE

    class ACL(ctypes.Structure):
        _fields_ = [('AclRevision', wintypes.BYTE), ('Sbz1', wintypes.BYTE), ('AclSize', wintypes.WORD), ('AceCount', wintypes.WORD), ('Sbz2', wintypes.WORD)]
    PSID = ctypes.c_void_p
    PACL = ctypes.POINTER(ACL)
    PSECURITY_DESCRIPTOR = ctypes.POINTER(wintypes.BYTE)

    def _nonzero_success(result: int, func: Any, args: Any) -> Any:
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return args
    secur32.GetUserNameExW.errcheck = _nonzero_success
    secur32.GetUserNameExW.restype = wintypes.BOOL
    secur32.GetUserNameExW.argtypes = (ctypes.c_int, wintypes.LPWSTR, wintypes.PULONG)
    advapi32.CreateWellKnownSid.errcheck = _nonzero_success
    advapi32.CreateWellKnownSid.restype = wintypes.BOOL
    advapi32.CreateWellKnownSid.argtypes = (wintypes.DWORD, PSID, PSID, wintypes.PDWORD)
    advapi32.LookupAccountNameW.errcheck = _nonzero_success
    advapi32.LookupAccountNameW.restype = wintypes.BOOL
    advapi32.LookupAccountNameW.argtypes = (wintypes.LPWSTR, wintypes.LPWSTR, PSID, wintypes.LPDWORD, wintypes.LPWSTR, wintypes.LPDWORD, wintypes.LPDWORD)
    advapi32.AddAccessAllowedAce.errcheck = _nonzero_success
    advapi32.AddAccessAllowedAce.restype = wintypes.BOOL
    advapi32.AddAccessAllowedAce.argtypes = (PACL, wintypes.DWORD, wintypes.DWORD, PSID)
    advapi32.SetSecurityDescriptorDacl.errcheck = _nonzero_success
    advapi32.SetSecurityDescriptorDacl.restype = wintypes.BOOL
    advapi32.SetSecurityDescriptorDacl.argtypes = (PSECURITY_DESCRIPTOR, wintypes.BOOL, PACL, wintypes.BOOL)
    advapi32.GetFileSecurityW.errcheck = _nonzero_success
    advapi32.GetFileSecurityW.restype = wintypes.BOOL
    advapi32.GetFileSecurityW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, PSECURITY_DESCRIPTOR, wintypes.DWORD, wintypes.LPDWORD)
    advapi32.SetFileSecurityW.errcheck = _nonzero_success
    advapi32.SetFileSecurityW.restype = wintypes.BOOL
    advapi32.SetFileSecurityW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, PSECURITY_DESCRIPTOR)
    advapi32.MakeAbsoluteSD.errcheck = _nonzero_success
    advapi32.MakeAbsoluteSD.restype = wintypes.BOOL
    advapi32.MakeAbsoluteSD.argtypes = (PSECURITY_DESCRIPTOR, PSECURITY_DESCRIPTOR, wintypes.LPDWORD, PACL, wintypes.LPDWORD, PACL, wintypes.LPDWORD, PSID, wintypes.LPDWORD, PSID, wintypes.LPDWORD)
    advapi32.MakeSelfRelativeSD.errcheck = _nonzero_success
    advapi32.MakeSelfRelativeSD.restype = wintypes.BOOL
    advapi32.MakeSelfRelativeSD.argtypes = (PSECURITY_DESCRIPTOR, PSECURITY_DESCRIPTOR, wintypes.LPDWORD)
    advapi32.InitializeAcl.errcheck = _nonzero_success
    advapi32.InitializeAcl.restype = wintypes.BOOL
    advapi32.InitializeAcl.argtypes = (PACL, wintypes.DWORD, wintypes.DWORD)

    def CreateWellKnownSid(WellKnownSidType: Any) -> Any:
        pSid = (ctypes.c_char * 1)()
        cbSid = wintypes.DWORD()
        try:
            advapi32.CreateWellKnownSid(WellKnownSidType, None, pSid, ctypes.byref(cbSid))
        except OSError as e:
            if e.winerror != ERROR_INSUFFICIENT_BUFFER:
                raise
            pSid = (ctypes.c_char * cbSid.value)()
            advapi32.CreateWellKnownSid(WellKnownSidType, None, pSid, ctypes.byref(cbSid))
        return pSid[:]

    def GetUserNameEx(NameFormat: Any) -> Any:
        nSize = ctypes.pointer(ctypes.c_ulong(0))
        try:
            secur32.GetUserNameExW(NameFormat, None, nSize)
        except OSError as e:
            if e.winerror != ERROR_MORE_DATA:
                raise
        if not nSize.contents.value:
            return None
        lpNameBuffer = ctypes.create_unicode_buffer(nSize.contents.value)
        secur32.GetUserNameExW(NameFormat, lpNameBuffer, nSize)
        return lpNameBuffer.value

    def LookupAccountName(lpSystemName: Any, lpAccountName: Any) -> Any:
        cbSid = wintypes.DWORD(0)
        cchReferencedDomainName = wintypes.DWORD(0)
        peUse = wintypes.DWORD(0)
        try:
            advapi32.LookupAccountNameW(lpSystemName, lpAccountName, None, ctypes.byref(cbSid), None, ctypes.byref(cchReferencedDomainName), ctypes.byref(peUse))
        except OSError as e:
            if e.winerror != ERROR_INSUFFICIENT_BUFFER:
                raise
        Sid = ctypes.create_unicode_buffer('', cbSid.value)
        pSid = ctypes.cast(ctypes.pointer(Sid), wintypes.LPVOID)
        lpReferencedDomainName = ctypes.create_unicode_buffer('', cchReferencedDomainName.value + 1)
        success = advapi32.LookupAccountNameW(lpSystemName, lpAccountName, pSid, ctypes.byref(cbSid), lpReferencedDomainName, ctypes.byref(cchReferencedDomainName), ctypes.byref(peUse))
        if not success:
            raise ctypes.WinError()
        return (pSid, lpReferencedDomainName.value, peUse.value)

    def AddAccessAllowedAce(pAcl: Any, dwAceRevision: Any, AccessMask: Any, pSid: Any) -> Any:
        advapi32.AddAccessAllowedAce(pAcl, dwAceRevision, AccessMask, pSid)

    def GetFileSecurity(lpFileName: Any, RequestedInformation: Any) -> Any:
        nLength = wintypes.DWORD(0)
        try:
            advapi32.GetFileSecurityW(lpFileName, RequestedInformation, None, 0, ctypes.byref(nLength))
        except OSError as e:
            if e.winerror != ERROR_INSUFFICIENT_BUFFER:
                raise
        if not nLength.value:
            return None
        pSecurityDescriptor = (wintypes.BYTE * nLength.value)()
        advapi32.GetFileSecurityW(lpFileName, RequestedInformation, pSecurityDescriptor, nLength, ctypes.byref(nLength))
        return pSecurityDescriptor

    def SetFileSecurity(lpFileName: Any, RequestedInformation: Any, pSecurityDescriptor: Any) -> Any:
        advapi32.SetFileSecurityW(lpFileName, RequestedInformation, pSecurityDescriptor)

    def SetSecurityDescriptorDacl(pSecurityDescriptor: Any, bDaclPresent: Any, pDacl: Any, bDaclDefaulted: Any) -> Any:
        advapi32.SetSecurityDescriptorDacl(pSecurityDescriptor, bDaclPresent, pDacl, bDaclDefaulted)

    def MakeAbsoluteSD(pSelfRelativeSecurityDescriptor: Any) -> Any:
        pAbsoluteSecurityDescriptor = None
        lpdwAbsoluteSecurityDescriptorSize = wintypes.DWORD(0)
        pDacl = None
        lpdwDaclSize = wintypes.DWORD(0)
        pSacl = None
        lpdwSaclSize = wintypes.DWORD(0)
        pOwner = None
        lpdwOwnerSize = wintypes.DWORD(0)
        pPrimaryGroup = None
        lpdwPrimaryGroupSize = wintypes.DWORD(0)
        try:
            advapi32.MakeAbsoluteSD(pSelfRelativeSecurityDescriptor, pAbsoluteSecurityDescriptor, ctypes.byref(lpdwAbsoluteSecurityDescriptorSize), pDacl, ctypes.byref(lpdwDaclSize), pSacl, ctypes.byref(lpdwSaclSize), pOwner, ctypes.byref(lpdwOwnerSize), pPrimaryGroup, ctypes.byref(lpdwPrimaryGroupSize))
        except OSError as e:
            if e.winerror != ERROR_INSUFFICIENT_BUFFER:
                raise
        pAbsoluteSecurityDescriptor = (wintypes.BYTE * lpdwAbsoluteSecurityDescriptorSize.value)()
        pDaclData = (wintypes.BYTE * lpdwDaclSize.value)()
        pDacl = ctypes.cast(pDaclData, PACL).contents
        pSaclData = (wintypes.BYTE * lpdwSaclSize.value)()
        pSacl = ctypes.cast(pSaclData, PACL).contents
        pOwnerData = (wintypes.BYTE * lpdwOwnerSize.value)()
        pOwner = ctypes.cast(pOwnerData, PSID)
        pPrimaryGroupData = (wintypes.BYTE * lpdwPrimaryGroupSize.value)()
        pPrimaryGroup = ctypes.cast(pPrimaryGroupData, PSID)
        advapi32.MakeAbsoluteSD(pSelfRelativeSecurityDescriptor, pAbsoluteSecurityDescriptor, ctypes.byref(lpdwAbsoluteSecurityDescriptorSize), pDacl, ctypes.byref(lpdwDaclSize), pSacl, ctypes.byref(lpdwSaclSize), pOwner, lpdwOwnerSize, pPrimaryGroup, ctypes.byref(lpdwPrimaryGroupSize))
        return pAbsoluteSecurityDescriptor

    def MakeSelfRelativeSD(pAbsoluteSecurityDescriptor: Any) -> Any:
        pSelfRelativeSecurityDescriptor = None
        lpdwBufferLength = wintypes.DWORD(0)
        try:
            advapi32.MakeSelfRelativeSD(pAbsoluteSecurityDescriptor, pSelfRelativeSecurityDescriptor, ctypes.byref(lpdwBufferLength))
        except OSError as e:
            if e.winerror != ERROR_INSUFFICIENT_BUFFER:
                raise
        pSelfRelativeSecurityDescriptor = (wintypes.BYTE * lpdwBufferLength.value)()
        advapi32.MakeSelfRelativeSD(pAbsoluteSecurityDescriptor, pSelfRelativeSecurityDescriptor, ctypes.byref(lpdwBufferLength))
        return pSelfRelativeSecurityDescriptor

    def NewAcl() -> Any:
        nAclLength = 32767
        acl_data = ctypes.create_string_buffer(nAclLength)
        pAcl = ctypes.cast(acl_data, PACL).contents
        advapi32.InitializeAcl(pAcl, nAclLength, ACL_REVISION)
        return pAcl
    SidAdmins = CreateWellKnownSid(WinBuiltinAdministratorsSid)
    SidUser = LookupAccountName('', GetUserNameEx(NameSamCompatible))[0]
    Acl = NewAcl()
    AddAccessAllowedAce(Acl, ACL_REVISION, FILE_ALL_ACCESS, SidAdmins)
    AddAccessAllowedAce(Acl, ACL_REVISION, FILE_GENERIC_READ | FILE_GENERIC_WRITE | DELETE, SidUser)
    SelfRelativeSD = GetFileSecurity(fname, DACL_SECURITY_INFORMATION)
    AbsoluteSD = MakeAbsoluteSD(SelfRelativeSD)
    SetSecurityDescriptorDacl(AbsoluteSD, 1, Acl, 0)
    SelfRelativeSD = MakeSelfRelativeSD(AbsoluteSD)
    SetFileSecurity(fname, DACL_SECURITY_INFORMATION, SelfRelativeSD)