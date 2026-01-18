from __future__ import (absolute_import, division, print_function)
import json
import os
import tarfile
from ansible.module_utils.common.text.converters import to_native
class ImageArchiveManifestSummary(object):
    """
    Represents data extracted from a manifest.json found in the tar archive output of the
    "docker image save some:tag > some.tar" command.
    """

    def __init__(self, image_id, repo_tags):
        """
        :param image_id:  File name portion of Config entry, e.g. abcde12345 from abcde12345.json
        :type image_id: str
        :param repo_tags  Docker image names, e.g. ["hello-world:latest"]
        :type repo_tags: list[str]
        """
        self.image_id = image_id
        self.repo_tags = repo_tags