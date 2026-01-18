import typing as tp
class BaseDataModel:

    def to_dict(self, calling_classes=None, recurse=False, render_unsets=False, **kwargs):
        """Converts a data model to a dictionary."""
        calling_classes = calling_classes or []
        ret = {}
        for attr, value in self.__dict__.items():
            if attr.startswith('_') or not kwargs.get(attr, True):
                continue
            if recurse:
                if isinstance(getattr(self, attr), list):
                    ret[attr] = []
                    for item in value:
                        if isinstance(item, BaseDataModel):
                            if type(self) not in calling_classes:
                                ret[attr].append(item.to_dict(calling_classes=calling_classes + [type(self)], recurse=True, render_unsets=render_unsets))
                            else:
                                ret[attr].append(None)
                        else:
                            ret[attr].append(item)
                elif isinstance(getattr(self, attr), BaseDataModel):
                    if type(self) not in calling_classes:
                        ret[attr] = value.to_dict(recurse=recurse, render_unsets=render_unsets, calling_classes=calling_classes + [type(self)])
                    else:
                        ret[attr] = None
                elif isinstance(value, UnsetType):
                    if render_unsets:
                        ret[attr] = None
                    else:
                        continue
                else:
                    ret[attr] = value
            elif isinstance(getattr(self, attr), (BaseDataModel, list)) or isinstance(value, UnsetType):
                if render_unsets:
                    ret[attr] = None
                else:
                    continue
            else:
                ret[attr] = value
        return ret

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.to_dict() == other.to_dict()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        attrs = ', '.join(('{}={!r}'.format(k, v) for k, v in sorted(self.__dict__.items())))
        return f'{self.__class__.__name__}({attrs})'

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)