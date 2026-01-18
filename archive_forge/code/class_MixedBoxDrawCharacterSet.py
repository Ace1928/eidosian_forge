from typing import List, NamedTuple, Optional
class MixedBoxDrawCharacterSet(_MixedBoxDrawCharacterSet):

    def char(self, *, top: int=0, bottom: int=0, left: int=0, right: int=0) -> Optional[str]:

        def parts_with(val: int) -> List[str]:
            parts = []
            if top == val:
                parts.append('top')
            if bottom == val:
                parts.append('bottom')
            if left == val:
                parts.append('left')
            if right == val:
                parts.append('right')
            return parts
        first_key = '_'.join(parts_with(-1))
        second_key = '_'.join(parts_with(+1))
        if not first_key and (not second_key):
            return None
        if not first_key:
            return getattr(self.second_char_set, second_key)
        if not second_key:
            return getattr(self.first_char_set, first_key)
        return getattr(self, f'{first_key}_then_{second_key}')