import copy
import logging
import re
from enum import Enum
from botocore import UNSIGNED, xform_name
from botocore.auth import AUTH_TYPE_MAPS, HAS_CRT
from botocore.crt import CRT_SUPPORTED_AUTH_TYPES
from botocore.endpoint_provider import EndpointProvider
from botocore.exceptions import (
from botocore.utils import ensure_boolean, instance_cache
class EndpointResolverBuiltins(str, Enum):
    AWS_REGION = 'AWS::Region'
    AWS_USE_FIPS = 'AWS::UseFIPS'
    AWS_USE_DUALSTACK = 'AWS::UseDualStack'
    AWS_STS_USE_GLOBAL_ENDPOINT = 'AWS::STS::UseGlobalEndpoint'
    AWS_S3_USE_GLOBAL_ENDPOINT = 'AWS::S3::UseGlobalEndpoint'
    AWS_S3_ACCELERATE = 'AWS::S3::Accelerate'
    AWS_S3_FORCE_PATH_STYLE = 'AWS::S3::ForcePathStyle'
    AWS_S3_USE_ARN_REGION = 'AWS::S3::UseArnRegion'
    AWS_S3CONTROL_USE_ARN_REGION = 'AWS::S3Control::UseArnRegion'
    AWS_S3_DISABLE_MRAP = 'AWS::S3::DisableMultiRegionAccessPoints'
    SDK_ENDPOINT = 'SDK::Endpoint'