from typing import List, Optional, Union
class SortbyField:

    def __init__(self, field: str, asc=True) -> None:
        self.args = [field, 'ASC' if asc else 'DESC']