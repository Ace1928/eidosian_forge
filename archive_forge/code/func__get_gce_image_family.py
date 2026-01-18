import re
import os.path
from typing import Dict, Union, Optional
from os.path import join as pjoin
def _get_gce_image_family(image_name):
    image_family = None
    if 'sql' in image_name:
        image_family = 'SQL Server'
    elif 'windows' in image_name:
        image_family = 'Windows Server'
    elif 'rhel' in image_name and 'sap' in image_name:
        image_family = 'RHEL with Update Services'
    elif 'sles' in image_name and 'sap' in image_name:
        image_family = 'SLES for SAP'
    elif 'rhel' in image_name:
        image_family = 'RHEL'
    elif 'sles' in image_name:
        image_family = 'SLES'
    return image_family