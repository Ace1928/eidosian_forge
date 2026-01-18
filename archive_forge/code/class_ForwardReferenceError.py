from typing import Any, Type, Optional, Set, Dict
class ForwardReferenceError(DaciteError):

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def __str__(self) -> str:
        return f'can not resolve forward reference: {self.message}'