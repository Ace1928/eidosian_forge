import warnings
import numpy as np
from ..utils import check_matplotlib_support
from ..utils._plotting import _interval_max_min_ratio, _validate_score_name
from ._validation import learning_curve, validation_curve
class _BaseCurveDisplay:

    def _plot_curve(self, x_data, *, ax=None, negate_score=False, score_name=None, score_type='test', log_scale='deprecated', std_display_style='fill_between', line_kw=None, fill_between_kw=None, errorbar_kw=None):
        check_matplotlib_support(f'{self.__class__.__name__}.plot')
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        if negate_score:
            train_scores, test_scores = (-self.train_scores, -self.test_scores)
        else:
            train_scores, test_scores = (self.train_scores, self.test_scores)
        if std_display_style not in ('errorbar', 'fill_between', None):
            raise ValueError(f"Unknown std_display_style: {std_display_style}. Should be one of 'errorbar', 'fill_between', or None.")
        if score_type not in ('test', 'train', 'both'):
            raise ValueError(f"Unknown score_type: {score_type}. Should be one of 'test', 'train', or 'both'.")
        if score_type == 'train':
            scores = {'Train': train_scores}
        elif score_type == 'test':
            scores = {'Test': test_scores}
        else:
            scores = {'Train': train_scores, 'Test': test_scores}
        if std_display_style in ('fill_between', None):
            if line_kw is None:
                line_kw = {}
            self.lines_ = []
            for line_label, score in scores.items():
                self.lines_.append(*ax.plot(x_data, score.mean(axis=1), label=line_label, **line_kw))
            self.errorbar_ = None
            self.fill_between_ = None
        if std_display_style == 'errorbar':
            if errorbar_kw is None:
                errorbar_kw = {}
            self.errorbar_ = []
            for line_label, score in scores.items():
                self.errorbar_.append(ax.errorbar(x_data, score.mean(axis=1), score.std(axis=1), label=line_label, **errorbar_kw))
            self.lines_, self.fill_between_ = (None, None)
        elif std_display_style == 'fill_between':
            if fill_between_kw is None:
                fill_between_kw = {}
            default_fill_between_kw = {'alpha': 0.5}
            fill_between_kw = {**default_fill_between_kw, **fill_between_kw}
            self.fill_between_ = []
            for line_label, score in scores.items():
                self.fill_between_.append(ax.fill_between(x_data, score.mean(axis=1) - score.std(axis=1), score.mean(axis=1) + score.std(axis=1), **fill_between_kw))
        score_name = self.score_name if score_name is None else score_name
        ax.legend()
        if log_scale != 'deprecated':
            warnings.warn('The `log_scale` parameter is deprecated as of version 1.3 and will be removed in 1.5. You can use display.ax_.set_xscale and display.ax_.set_yscale instead.', FutureWarning)
            xscale = 'log' if log_scale else 'linear'
        elif _interval_max_min_ratio(x_data) > 5:
            xscale = 'symlog' if x_data.min() <= 0 else 'log'
        else:
            xscale = 'linear'
        ax.set_xscale(xscale)
        ax.set_ylabel(f'{score_name}')
        self.ax_ = ax
        self.figure_ = ax.figure