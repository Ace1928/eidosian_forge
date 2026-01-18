from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
def _assert_exec_policy_type(exec_policy: _Data):
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise TypeError(f'{exec_policy.code} must be execution policy type')