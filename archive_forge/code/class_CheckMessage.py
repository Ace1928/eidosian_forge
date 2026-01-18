class CheckMessage:

    def __init__(self, level, msg, hint=None, obj=None, id=None):
        if not isinstance(level, int):
            raise TypeError('The first argument should be level.')
        self.level = level
        self.msg = msg
        self.hint = hint
        self.obj = obj
        self.id = id

    def __eq__(self, other):
        return isinstance(other, self.__class__) and all((getattr(self, attr) == getattr(other, attr) for attr in ['level', 'msg', 'hint', 'obj', 'id']))

    def __str__(self):
        from django.db import models
        if self.obj is None:
            obj = '?'
        elif isinstance(self.obj, models.base.ModelBase):
            obj = self.obj._meta.label
        else:
            obj = str(self.obj)
        id = '(%s) ' % self.id if self.id else ''
        hint = '\n\tHINT: %s' % self.hint if self.hint else ''
        return '%s: %s%s%s' % (obj, id, self.msg, hint)

    def __repr__(self):
        return '<%s: level=%r, msg=%r, hint=%r, obj=%r, id=%r>' % (self.__class__.__name__, self.level, self.msg, self.hint, self.obj, self.id)

    def is_serious(self, level=ERROR):
        return self.level >= level

    def is_silenced(self):
        from django.conf import settings
        return self.id in settings.SILENCED_SYSTEM_CHECKS