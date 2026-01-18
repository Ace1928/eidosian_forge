from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InfiniteLoopConfigError(InvalidConfigError):
    fmt = 'Infinite loop in credential configuration detected. Attempting to load from profile {source_profile} which has already been visited. Visited profiles: {visited_profiles}'