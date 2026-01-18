import os
def ShellifyStatus(status):
    """Translate from a wait() exit status to a command shell exit status."""
    if not win32:
        if os.WIFEXITED(status):
            status = os.WEXITSTATUS(status)
        else:
            status = 128 + os.WTERMSIG(status)
    return status