from collections import namedtuple
from threading import Lock
from ._authorizer import ClosedAuthorizer
from ._identity import NoIdentities
from ._error import (
import macaroonbakery.checkers as checkers
import pyrfc3339
def _allow_any(self, ctx, ops):
    self._init(ctx)
    used = [False] * len(self._macaroons)
    authed = [False] * len(ops)
    num_authed = 0
    errors = []
    for i, op in enumerate(ops):
        for mindex in self._auth_indexes.get(op, []):
            _, err = self._check_conditions(ctx, op, self._conditions[mindex])
            if err is not None:
                errors.append(err)
                continue
            authed[i] = True
            num_authed += 1
            used[mindex] = True
            break
        if op == LOGIN_OP and (not authed[i]) and (self._identity is not None):
            authed[i] = True
    if self._identity is not None:
        for i in self._auth_indexes.get(LOGIN_OP, []):
            used[i] = True
    if num_authed == len(ops):
        return (authed, used)
    need = []
    need_index = [0] * (len(ops) - num_authed)
    for i, ok in enumerate(authed):
        if not ok:
            need_index[len(need)] = i
            need.append(ops[i])
    oks, caveats = self.parent._authorizer.authorize(ctx, self._identity, need)
    still_need = []
    for i, _ in enumerate(need):
        if i < len(oks) and oks[i]:
            authed[need_index[i]] = True
        else:
            still_need.append(ops[need_index[i]])
    if len(still_need) == 0 and len(caveats) == 0:
        return (authed, used)
    if self._identity is None and len(self._identity_caveats) > 0:
        raise DischargeRequiredError(msg='authentication required', ops=[LOGIN_OP], cavs=self._identity_caveats)
    if caveats is None or len(caveats) == 0:
        all_errors = []
        all_errors.extend(self._init_errors)
        all_errors.extend(errors)
        err = ''
        if len(all_errors) > 0:
            err = all_errors[0]
        raise PermissionDenied(err)
    raise DischargeRequiredError(msg='some operations have extra caveats', ops=ops, cavs=caveats)