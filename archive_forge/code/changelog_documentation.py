import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
 Write the changelog entry to a filehandle

        Write the changelog out to the filehandle passed. The file argument
        must be an open file object.
        