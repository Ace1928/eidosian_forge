from math import ceil
import cupy
def _get_max_gdx():
    device_id = cupy.cuda.Device()
    return device_id.attributes['MaxGridDimX']