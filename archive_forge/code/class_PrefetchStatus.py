from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
class PrefetchStatus(enum.Enum):
    """
    TODO(https://crbug.com/1384419): revisit the list of PrefetchStatus and
    filter out the ones that aren't necessary to the developers.
    """
    PREFETCH_ALLOWED = 'PrefetchAllowed'
    PREFETCH_FAILED_INELIGIBLE_REDIRECT = 'PrefetchFailedIneligibleRedirect'
    PREFETCH_FAILED_INVALID_REDIRECT = 'PrefetchFailedInvalidRedirect'
    PREFETCH_FAILED_MIME_NOT_SUPPORTED = 'PrefetchFailedMIMENotSupported'
    PREFETCH_FAILED_NET_ERROR = 'PrefetchFailedNetError'
    PREFETCH_FAILED_NON2_XX = 'PrefetchFailedNon2XX'
    PREFETCH_FAILED_PER_PAGE_LIMIT_EXCEEDED = 'PrefetchFailedPerPageLimitExceeded'
    PREFETCH_EVICTED_AFTER_CANDIDATE_REMOVED = 'PrefetchEvictedAfterCandidateRemoved'
    PREFETCH_EVICTED_FOR_NEWER_PREFETCH = 'PrefetchEvictedForNewerPrefetch'
    PREFETCH_HELDBACK = 'PrefetchHeldback'
    PREFETCH_INELIGIBLE_RETRY_AFTER = 'PrefetchIneligibleRetryAfter'
    PREFETCH_IS_PRIVACY_DECOY = 'PrefetchIsPrivacyDecoy'
    PREFETCH_IS_STALE = 'PrefetchIsStale'
    PREFETCH_NOT_ELIGIBLE_BROWSER_CONTEXT_OFF_THE_RECORD = 'PrefetchNotEligibleBrowserContextOffTheRecord'
    PREFETCH_NOT_ELIGIBLE_DATA_SAVER_ENABLED = 'PrefetchNotEligibleDataSaverEnabled'
    PREFETCH_NOT_ELIGIBLE_EXISTING_PROXY = 'PrefetchNotEligibleExistingProxy'
    PREFETCH_NOT_ELIGIBLE_HOST_IS_NON_UNIQUE = 'PrefetchNotEligibleHostIsNonUnique'
    PREFETCH_NOT_ELIGIBLE_NON_DEFAULT_STORAGE_PARTITION = 'PrefetchNotEligibleNonDefaultStoragePartition'
    PREFETCH_NOT_ELIGIBLE_SAME_SITE_CROSS_ORIGIN_PREFETCH_REQUIRED_PROXY = 'PrefetchNotEligibleSameSiteCrossOriginPrefetchRequiredProxy'
    PREFETCH_NOT_ELIGIBLE_SCHEME_IS_NOT_HTTPS = 'PrefetchNotEligibleSchemeIsNotHttps'
    PREFETCH_NOT_ELIGIBLE_USER_HAS_COOKIES = 'PrefetchNotEligibleUserHasCookies'
    PREFETCH_NOT_ELIGIBLE_USER_HAS_SERVICE_WORKER = 'PrefetchNotEligibleUserHasServiceWorker'
    PREFETCH_NOT_ELIGIBLE_BATTERY_SAVER_ENABLED = 'PrefetchNotEligibleBatterySaverEnabled'
    PREFETCH_NOT_ELIGIBLE_PRELOADING_DISABLED = 'PrefetchNotEligiblePreloadingDisabled'
    PREFETCH_NOT_FINISHED_IN_TIME = 'PrefetchNotFinishedInTime'
    PREFETCH_NOT_STARTED = 'PrefetchNotStarted'
    PREFETCH_NOT_USED_COOKIES_CHANGED = 'PrefetchNotUsedCookiesChanged'
    PREFETCH_PROXY_NOT_AVAILABLE = 'PrefetchProxyNotAvailable'
    PREFETCH_RESPONSE_USED = 'PrefetchResponseUsed'
    PREFETCH_SUCCESSFUL_BUT_NOT_USED = 'PrefetchSuccessfulButNotUsed'
    PREFETCH_NOT_USED_PROBE_FAILED = 'PrefetchNotUsedProbeFailed'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)