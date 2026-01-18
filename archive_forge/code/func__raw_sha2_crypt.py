import hashlib
import logging; log = logging.getLogger(__name__)
from passlib.utils import safe_crypt, test_crypt, \
from passlib.utils.binary import h64
from passlib.utils.compat import byte_elem_value, u, \
import passlib.utils.handlers as uh
def _raw_sha2_crypt(pwd, salt, rounds, use_512=False):
    """perform raw sha256-crypt / sha512-crypt

    this function provides a pure-python implementation of the internals
    for the SHA256-Crypt and SHA512-Crypt algorithms; it doesn't
    handle any of the parsing/validation of the hash strings themselves.

    :arg pwd: password chars/bytes to hash
    :arg salt: salt chars to use
    :arg rounds: linear rounds cost
    :arg use_512: use sha512-crypt instead of sha256-crypt mode

    :returns:
        encoded checksum chars
    """
    if isinstance(pwd, unicode):
        pwd = pwd.encode('utf-8')
    assert isinstance(pwd, bytes)
    if _BNULL in pwd:
        raise uh.exc.NullPasswordError(sha512_crypt if use_512 else sha256_crypt)
    pwd_len = len(pwd)
    assert 1000 <= rounds <= 999999999, 'invalid rounds'
    assert isinstance(salt, unicode), 'salt not unicode'
    salt = salt.encode('ascii')
    salt_len = len(salt)
    assert salt_len < 17, 'salt too large'
    if use_512:
        hash_const = hashlib.sha512
        transpose_map = _512_transpose_map
    else:
        hash_const = hashlib.sha256
        transpose_map = _256_transpose_map
    db = hash_const(pwd + salt + pwd).digest()
    a_ctx = hash_const(pwd + salt)
    a_ctx_update = a_ctx.update
    a_ctx_update(repeat_string(db, pwd_len))
    i = pwd_len
    while i:
        a_ctx_update(db if i & 1 else pwd)
        i >>= 1
    da = a_ctx.digest()
    if pwd_len < 96:
        dp = repeat_string(hash_const(pwd * pwd_len).digest(), pwd_len)
    else:
        tmp_ctx = hash_const(pwd)
        tmp_ctx_update = tmp_ctx.update
        i = pwd_len - 1
        while i:
            tmp_ctx_update(pwd)
            i -= 1
        dp = repeat_string(tmp_ctx.digest(), pwd_len)
    assert len(dp) == pwd_len
    ds = hash_const(salt * (16 + byte_elem_value(da[0]))).digest()[:salt_len]
    assert len(ds) == salt_len, 'salt_len somehow > hash_len!'
    dp_dp = dp + dp
    dp_ds = dp + ds
    perms = [dp, dp_dp, dp_ds, dp_ds + dp, ds + dp, ds + dp_dp]
    data = [(perms[even], perms[odd]) for even, odd in _c_digest_offsets]
    dc = da
    blocks, tail = divmod(rounds, 42)
    while blocks:
        for even, odd in data:
            dc = hash_const(odd + hash_const(dc + even).digest()).digest()
        blocks -= 1
    if tail:
        pairs = tail >> 1
        for even, odd in data[:pairs]:
            dc = hash_const(odd + hash_const(dc + even).digest()).digest()
        if tail & 1:
            dc = hash_const(dc + data[pairs][0]).digest()
    return h64.encode_transposed_bytes(dc, transpose_map).decode('ascii')