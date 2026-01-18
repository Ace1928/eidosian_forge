from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
from boto import config
import gslib
from gslib import exception
from gslib.utils import boto_util
from gslib.utils import execution_util
def _split_pem_into_sections(contents, logger):
    """Returns dict with {name: section} by parsing contents in PEM format.

  A simple parser for PEM file. Please see RFC 7468 for the format of PEM
  file. Not using regex to improve performance catching nested matches.
  Note: This parser requires the post-encapsulation label of a section to
  match its pre-encapsulation label. It ignores a section without a
  matching label.

  Args:
    contents (str): Contents of a PEM file.
    logger (logging.logger): gsutil logger.

  Returns:
    A dict of the PEM file sections.
  """
    result = {}
    pem_lines = []
    pem_section_name = None
    for line in contents.splitlines():
        line = line.strip()
        if not line:
            continue
        begin, end, name = _is_pem_section_marker(line)
        if begin:
            if pem_section_name:
                logger.warning('Section %s missing end line and will be ignored.' % pem_section_name)
            if name in result.keys():
                logger.warning('Section %s already exists, and the older section will be ignored.' % name)
            pem_section_name = name
            pem_lines = []
        elif end:
            if not pem_section_name:
                logger.warning('Section %s missing a beginning line and will be ignored.' % name)
            elif pem_section_name != name:
                logger.warning('Section %s missing a matching end line. Found: %s' % (pem_section_name, name))
                pem_section_name = None
        if pem_section_name:
            pem_lines.append(line)
            if end:
                result[name] = '\n'.join(pem_lines) + '\n'
                pem_section_name = None
    if pem_section_name:
        logger.warning('Section %s missing an end line.' % pem_section_name)
    return result