from django.db.models import Aggregate, FloatField, IntegerField
class CovarPop(StatAggregate):

    def __init__(self, y, x, sample=False, filter=None, default=None):
        self.function = 'COVAR_SAMP' if sample else 'COVAR_POP'
        super().__init__(y, x, filter=filter, default=default)