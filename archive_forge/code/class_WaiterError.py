from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class WaiterError(BotoCoreError):
    """Waiter failed to reach desired state."""
    fmt = 'Waiter {name} failed: {reason}'

    def __init__(self, name, reason, last_response):
        super().__init__(name=name, reason=reason)
        self.last_response = last_response