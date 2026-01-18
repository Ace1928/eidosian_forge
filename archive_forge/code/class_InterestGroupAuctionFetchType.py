from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
class InterestGroupAuctionFetchType(enum.Enum):
    """
    Enum of network fetches auctions can do.
    """
    BIDDER_JS = 'bidderJs'
    BIDDER_WASM = 'bidderWasm'
    SELLER_JS = 'sellerJs'
    BIDDER_TRUSTED_SIGNALS = 'bidderTrustedSignals'
    SELLER_TRUSTED_SIGNALS = 'sellerTrustedSignals'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)