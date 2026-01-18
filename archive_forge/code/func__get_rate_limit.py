from datetime import datetime
from oslo_utils import timeutils
def _get_rate_limit(self, resp):
    if resp is not None and resp.headers:
        utc_now = timeutils.utcnow()
        value = resp.headers.get('Retry-After', '0')
        try:
            value = datetime.strptime(value, '%a, %d %b %Y %H:%M:%S %Z')
            if value > utc_now:
                self.retry_after = (value - utc_now).seconds
            else:
                self.retry_after = 0
        except ValueError:
            self.retry_after = int(value)