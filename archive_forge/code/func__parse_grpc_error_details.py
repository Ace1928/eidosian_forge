from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
def _parse_grpc_error_details(rpc_exc):
    try:
        status = rpc_status.from_call(rpc_exc)
    except NotImplementedError:
        return ([], None)
    if not status:
        return ([], None)
    possible_errors = [error_details_pb2.BadRequest, error_details_pb2.PreconditionFailure, error_details_pb2.QuotaFailure, error_details_pb2.ErrorInfo, error_details_pb2.RetryInfo, error_details_pb2.ResourceInfo, error_details_pb2.RequestInfo, error_details_pb2.DebugInfo, error_details_pb2.Help, error_details_pb2.LocalizedMessage]
    error_info = None
    error_details = []
    for detail in status.details:
        matched_detail_cls = list(filter(lambda x: detail.Is(x.DESCRIPTOR), possible_errors))
        if len(matched_detail_cls) == 0:
            info = detail
        else:
            info = matched_detail_cls[0]()
            detail.Unpack(info)
        error_details.append(info)
        if isinstance(info, error_details_pb2.ErrorInfo):
            error_info = info
    return (error_details, error_info)