from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class InterestGroupDetails:
    """
    The full details of an interest group.
    """
    owner_origin: str
    name: str
    expiration_time: network.TimeSinceEpoch
    joining_origin: str
    trusted_bidding_signals_keys: typing.List[str]
    ads: typing.List[InterestGroupAd]
    ad_components: typing.List[InterestGroupAd]
    bidding_logic_url: typing.Optional[str] = None
    bidding_wasm_helper_url: typing.Optional[str] = None
    update_url: typing.Optional[str] = None
    trusted_bidding_signals_url: typing.Optional[str] = None
    user_bidding_signals: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['ownerOrigin'] = self.owner_origin
        json['name'] = self.name
        json['expirationTime'] = self.expiration_time.to_json()
        json['joiningOrigin'] = self.joining_origin
        json['trustedBiddingSignalsKeys'] = [i for i in self.trusted_bidding_signals_keys]
        json['ads'] = [i.to_json() for i in self.ads]
        json['adComponents'] = [i.to_json() for i in self.ad_components]
        if self.bidding_logic_url is not None:
            json['biddingLogicURL'] = self.bidding_logic_url
        if self.bidding_wasm_helper_url is not None:
            json['biddingWasmHelperURL'] = self.bidding_wasm_helper_url
        if self.update_url is not None:
            json['updateURL'] = self.update_url
        if self.trusted_bidding_signals_url is not None:
            json['trustedBiddingSignalsURL'] = self.trusted_bidding_signals_url
        if self.user_bidding_signals is not None:
            json['userBiddingSignals'] = self.user_bidding_signals
        return json

    @classmethod
    def from_json(cls, json):
        return cls(owner_origin=str(json['ownerOrigin']), name=str(json['name']), expiration_time=network.TimeSinceEpoch.from_json(json['expirationTime']), joining_origin=str(json['joiningOrigin']), trusted_bidding_signals_keys=[str(i) for i in json['trustedBiddingSignalsKeys']], ads=[InterestGroupAd.from_json(i) for i in json['ads']], ad_components=[InterestGroupAd.from_json(i) for i in json['adComponents']], bidding_logic_url=str(json['biddingLogicURL']) if 'biddingLogicURL' in json else None, bidding_wasm_helper_url=str(json['biddingWasmHelperURL']) if 'biddingWasmHelperURL' in json else None, update_url=str(json['updateURL']) if 'updateURL' in json else None, trusted_bidding_signals_url=str(json['trustedBiddingSignalsURL']) if 'trustedBiddingSignalsURL' in json else None, user_bidding_signals=str(json['userBiddingSignals']) if 'userBiddingSignals' in json else None)