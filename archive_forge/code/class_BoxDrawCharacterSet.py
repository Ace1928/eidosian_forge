from typing import List, NamedTuple, Optional
class BoxDrawCharacterSet(_BoxDrawCharacterSet):

    def char(self, top: bool=False, bottom: bool=False, left: bool=False, right: bool=False) -> Optional[str]:
        parts = []
        if top:
            parts.append('top')
        if bottom:
            parts.append('bottom')
        if left:
            parts.append('left')
        if right:
            parts.append('right')
        if not parts:
            return None
        return getattr(self, '_'.join(parts))