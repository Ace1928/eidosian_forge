import sys
import sysconfig
def _aix_bgt():
    gnu_type = sysconfig.get_config_var('BUILD_GNU_TYPE')
    if not gnu_type:
        raise ValueError('BUILD_GNU_TYPE is not defined')
    return _aix_vrtl(vrmf=gnu_type)