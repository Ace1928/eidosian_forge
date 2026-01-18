from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@event_class('Storage.interestGroupAuctionEventOccurred')
@dataclass
class InterestGroupAuctionEventOccurred:
    """
    An auction involving interest groups is taking place. These events are
    target-specific.
    """
    event_time: network.TimeSinceEpoch
    type_: InterestGroupAuctionEventType
    unique_auction_id: InterestGroupAuctionId
    parent_auction_id: typing.Optional[InterestGroupAuctionId]
    auction_config: typing.Optional[dict]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterestGroupAuctionEventOccurred:
        return cls(event_time=network.TimeSinceEpoch.from_json(json['eventTime']), type_=InterestGroupAuctionEventType.from_json(json['type']), unique_auction_id=InterestGroupAuctionId.from_json(json['uniqueAuctionId']), parent_auction_id=InterestGroupAuctionId.from_json(json['parentAuctionId']) if 'parentAuctionId' in json else None, auction_config=dict(json['auctionConfig']) if 'auctionConfig' in json else None)