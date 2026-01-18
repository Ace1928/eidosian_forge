from ..base import (
import os
def _get_cleaned_functional_filename(self, artifacts_list_filename):
    """extract the proper filename from the first line of the artifacts file"""
    artifacts_list_file = open(artifacts_list_filename, 'r')
    functional_filename, extension = artifacts_list_file.readline().split('.')
    artifacts_list_file_path, artifacts_list_filename = os.path.split(artifacts_list_filename)
    return os.path.join(artifacts_list_file_path, functional_filename + '_clean.nii.gz')