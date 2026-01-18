import re
import os.path
from typing import Dict, Union, Optional
from os.path import join as pjoin
def _get_gce_image_price(image_name, size_name, cores=1):
    """
    Return price per hour for an gce image.
    Price depends on the size of the VM.

    :type image_name: ``str``
    :param image_name: GCE image full name.
                       Can be found from GCENodeImage.name

    :type size_name: ``str``
    :param size_name: Size name of the machine running the image.
                      Can be found from GCENodeSize.name

    :type cores: ``int``
    :param cores: The number of the CPUs the machine running the image has.
                  Can be found from GCENodeSize.extra['guestCpus']

    :rtype: ``float``
    :return: Image price
    """

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
    image_family = _get_gce_image_family(image_name)
    if not image_family:
        return 0
    pricing = get_pricing(driver_type='compute', driver_name='gce_images')
    try:
        price_dict = pricing[image_family]
    except KeyError:
        return 0
    size_type = 'any'
    if 'f1' in size_name:
        size_type = 'f1'
    elif 'g1' in size_name:
        size_type = 'g1'
    price_dict_keys = price_dict.keys()
    for key in price_dict_keys:
        if key == 'description':
            continue
        if re.search('.{1}vcpu or less', key) and cores <= int(key[0]):
            return float(price_dict[key]['price'])
        if re.search('.{1}-.{1}vcpu', key) and str(cores) in key:
            return float(price_dict[key]['price'])
        if re.search('.{1}vcpu or more', key) and cores >= int(key[0]):
            return float(price_dict[key]['price'])
        if key in {'standard', 'enterprise', 'web'} and key in image_name:
            return float(price_dict[key]['price'])
        if key in {'f1', 'g1'} and size_type == key:
            return float(price_dict[key]['price'])
        elif key == 'any':
            price = float(price_dict[key]['price'])
            return price * cores if 'sles' not in image_name else price
    return 0