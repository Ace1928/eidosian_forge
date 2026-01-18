from typing import Any, Dict, Iterable
Split a file's dict (see `files` arg) into smaller dicts.

    Each smaller dict will have at most `MAX_BYTES` size.

    This method is used in `FileStreamAPI._send()` to limit the size of post requests
    sent to wandb server.

    Arguments:
    files (dict): `dict` of form {file_name: {'content': ".....", 'offset': 0}}
        The key `file_name` can also be mapped to a List [{"offset": int, "content": str}]
    `max_bytes`: max size for chunk in bytes
    