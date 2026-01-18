import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
def _prepare_send_function(self):
    """Write our wrapper function into a temporary file.

        This temporary file will be loaded at runtime in
        _get_compose_commandline function.

        This function does not remove the file.  That's a wanted
        behaviour since _get_compose_commandline won't run the send
        mail function directly but return the eligible command line.
        Removing our temporary file here would prevent our sendmail
        function to work.  (The file is deleted by some elisp code
        after being read by Emacs.)
        """
    _defun = b'(defun bzr-add-mime-att (file)\n  "Attach FILE to a mail buffer as a MIME attachment."\n  (let ((agent mail-user-agent))\n    (if (and file (file-exists-p file))\n        (cond\n         ((eq agent \'sendmail-user-agent)\n          (progn\n            (mail-text)\n            (newline)\n            (if (functionp \'etach-attach)\n              (etach-attach file)\n              (mail-attach-file file))))\n         ((or (eq agent \'message-user-agent)\n              (eq agent \'gnus-user-agent)\n              (eq agent \'mh-e-user-agent))\n          (progn\n            (mml-attach-file file "text/x-patch" "BZR merge" "inline")))\n         ((eq agent \'mew-user-agent)\n          (progn\n            (mew-draft-prepare-attachments)\n            (mew-attach-link file (file-name-nondirectory file))\n            (let* ((nums (mew-syntax-nums))\n                   (syntax (mew-syntax-get-entry mew-encode-syntax nums)))\n              (mew-syntax-set-cd syntax "BZR merge")\n              (mew-encode-syntax-print mew-encode-syntax))\n            (mew-header-goto-body)))\n         (t\n          (message "Unhandled MUA, report it on bazaar@lists.canonical.com")))\n      (error "File %s does not exist." file))))\n'
    fd, temp_file = tempfile.mkstemp(prefix='emacs-bzr-send-', suffix='.el')
    try:
        os.write(fd, _defun)
    finally:
        os.close(fd)
    return temp_file