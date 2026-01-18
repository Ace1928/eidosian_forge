import abc
import typing as t
from .interface.summary_record import SummaryItem, SummaryRecord
def _get_dict(d):
    if isinstance(d, dict):
        return d
    return vars(d)