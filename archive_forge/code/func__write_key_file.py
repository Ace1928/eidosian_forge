import datetime
import glob
import os
from typing import Dict, Optional, Tuple, Union
import zmq
def _write_key_file(key_filename: Union[str, os.PathLike], banner: str, public_key: Union[str, bytes], secret_key: Optional[Union[str, bytes]]=None, metadata: Optional[Dict[str, str]]=None, encoding: str='utf-8') -> None:
    """Create a certificate file"""
    if isinstance(public_key, bytes):
        public_key = public_key.decode(encoding)
    if isinstance(secret_key, bytes):
        secret_key = secret_key.decode(encoding)
    with open(key_filename, 'w', encoding='utf8') as f:
        f.write(banner.format(datetime.datetime.now()))
        f.write('metadata\n')
        if metadata:
            for k, v in metadata.items():
                if isinstance(k, bytes):
                    k = k.decode(encoding)
                if isinstance(v, bytes):
                    v = v.decode(encoding)
                f.write(f'    {k} = {v}\n')
        f.write('curve\n')
        f.write(f'    public-key = "{public_key}"\n')
        if secret_key:
            f.write(f'    secret-key = "{secret_key}"\n')