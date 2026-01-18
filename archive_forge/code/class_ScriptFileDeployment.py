import os
import re
import binascii
from typing import IO, List, Union, Optional, cast
from libcloud.utils.py3 import basestring
from libcloud.compute.ssh import BaseSSHClient
from libcloud.compute.base import Node
class ScriptFileDeployment(ScriptDeployment):
    """
    Runs an arbitrary shell script from a local file on the server. Same as
    ScriptDeployment, except that you can pass in a path to the file instead of
    the script content.
    """

    def __init__(self, script_file, args=None, name=None, delete=False, timeout=None):
        """
        :type script_file: ``str``
        :keyword script_file: Path to a file containing the script to run.

        :type args: ``list``
        :keyword args: Optional command line arguments which get passed to the
                       deployment script file.


        :type name: ``str``
        :keyword name: Name of the script to upload it as, if not specified,
                       a random name will be chosen.

        :type delete: ``bool``
        :keyword delete: Whether to delete the script on completion.

        :param timeout: Optional run timeout for this command.
        :type timeout: ``float``
        """
        with open(script_file, 'rb') as fp:
            content = fp.read()
        content = cast(bytes, content)
        content = content.decode('utf-8')
        super().__init__(script=content, args=args, name=name, delete=delete, timeout=timeout)