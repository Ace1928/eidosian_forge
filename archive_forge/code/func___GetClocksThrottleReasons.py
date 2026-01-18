from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __GetClocksThrottleReasons(handle):
    throttleReasons = [[nvmlClocksThrottleReasonGpuIdle, 'clocks_throttle_reason_gpu_idle'], [nvmlClocksThrottleReasonUserDefinedClocks, 'clocks_throttle_reason_user_defined_clocks'], [nvmlClocksThrottleReasonApplicationsClocksSetting, 'clocks_throttle_reason_applications_clocks_setting'], [nvmlClocksThrottleReasonSwPowerCap, 'clocks_throttle_reason_sw_power_cap'], [nvmlClocksThrottleReasonHwSlowdown, 'clocks_throttle_reason_hw_slowdown'], [nvmlClocksThrottleReasonNone, 'clocks_throttle_reason_none']]
    clockThrottleReasons = {}
    try:
        supportedClocksThrottleReasons = nvmlDeviceGetSupportedClocksThrottleReasons(handle)
        clocksThrottleReasons = nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        for mask, name in throttleReasons:
            if name != 'clocks_throttle_reason_user_defined_clocks':
                if mask & supportedClocksThrottleReasons:
                    val = 'Active' if mask & clocksThrottleReasons else 'Not Active'
                else:
                    val = 'N/A'
                clockThrottleReasons[name] = val
    except NVMLError as err:
        clockThrottleReasons['Error'] = nvidia_smi.__handleError(err)
    return clockThrottleReasons if len(clockThrottleReasons.values()) > 0 else None