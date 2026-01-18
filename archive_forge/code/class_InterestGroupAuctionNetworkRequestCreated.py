from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@event_class('Storage.interestGroupAuctionNetworkRequestCreated')
@dataclass
class InterestGroupAuctionNetworkRequestCreated:
    """
    Specifies which auctions a particular network fetch may be related to, and
    in what role. Note that it is not ordered with respect to
    Network.requestWillBeSent (but will happen before loadingFinished
    loadingFailed).
    """
    type_: InterestGroupAuctionFetchType
    request_id: network.RequestId
    auctions: typing.List[InterestGroupAuctionId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterestGroupAuctionNetworkRequestCreated:
        return cls(type_=InterestGroupAuctionFetchType.from_json(json['type']), request_id=network.RequestId.from_json(json['requestId']), auctions=[InterestGroupAuctionId.from_json(i) for i in json['auctions']])