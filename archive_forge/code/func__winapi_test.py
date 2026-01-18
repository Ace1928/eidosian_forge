def _winapi_test(handle):
    csbi = CONSOLE_SCREEN_BUFFER_INFO()
    success = _GetConsoleScreenBufferInfo(handle, byref(csbi))
    return bool(success)