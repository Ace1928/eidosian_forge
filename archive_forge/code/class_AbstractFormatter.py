import re
import html
from paste.util import PySourceColor
class AbstractFormatter(object):
    general_data_order = ['object', 'source_url']

    def __init__(self, show_hidden_frames=False, include_reusable=True, show_extra_data=True, trim_source_paths=()):
        self.show_hidden_frames = show_hidden_frames
        self.trim_source_paths = trim_source_paths
        self.include_reusable = include_reusable
        self.show_extra_data = show_extra_data

    def format_collected_data(self, exc_data):
        general_data = {}
        if self.show_extra_data:
            for name, value_list in exc_data.extra_data.items():
                if isinstance(name, tuple):
                    importance, title = name
                else:
                    importance, title = ('normal', name)
                for value in value_list:
                    general_data[importance, name] = self.format_extra_data(importance, title, value)
        lines = []
        frames = self.filter_frames(exc_data.frames)
        for frame in frames:
            sup = frame.supplement
            if sup:
                if sup.object:
                    general_data['important', 'object'] = self.format_sup_object(sup.object)
                if sup.source_url:
                    general_data['important', 'source_url'] = self.format_sup_url(sup.source_url)
                if sup.line:
                    lines.append(self.format_sup_line_pos(sup.line, sup.column))
                if sup.expression:
                    lines.append(self.format_sup_expression(sup.expression))
                if sup.warnings:
                    for warning in sup.warnings:
                        lines.append(self.format_sup_warning(warning))
                if sup.info:
                    lines.extend(self.format_sup_info(sup.info))
            if frame.supplement_exception:
                lines.append('Exception in supplement:')
                lines.append(self.quote_long(frame.supplement_exception))
            if frame.traceback_info:
                lines.append(self.format_traceback_info(frame.traceback_info))
            filename = frame.filename
            if filename and self.trim_source_paths:
                for path, repl in self.trim_source_paths:
                    if filename.startswith(path):
                        filename = repl + filename[len(path):]
                        break
            lines.append(self.format_source_line(filename or '?', frame))
            source = frame.get_source_line()
            long_source = frame.get_source_line(2)
            if source:
                lines.append(self.format_long_source(source, long_source))
        etype = exc_data.exception_type
        if not isinstance(etype, str):
            etype = etype.__name__
        exc_info = self.format_exception_info(etype, exc_data.exception_value)
        data_by_importance = {'important': [], 'normal': [], 'supplemental': [], 'extra': []}
        for (importance, name), value in general_data.items():
            data_by_importance[importance].append((name, value))
        for value in data_by_importance.values():
            value.sort()
        return self.format_combine(data_by_importance, lines, exc_info)

    def filter_frames(self, frames):
        """
        Removes any frames that should be hidden, according to the
        values of traceback_hide, self.show_hidden_frames, and the
        hidden status of the final frame.
        """
        if self.show_hidden_frames:
            return frames
        new_frames = []
        hidden = False
        for frame in frames:
            hide = frame.traceback_hide
            if hide == 'before':
                new_frames = []
                hidden = False
            elif hide == 'before_and_this':
                new_frames = []
                hidden = False
                continue
            elif hide == 'reset':
                hidden = False
            elif hide == 'reset_and_this':
                hidden = False
                continue
            elif hide == 'after':
                hidden = True
            elif hide == 'after_and_this':
                hidden = True
                continue
            elif hide:
                continue
            elif hidden:
                continue
            new_frames.append(frame)
        if frames[-1] not in new_frames:
            return frames
        return new_frames

    def pretty_string_repr(self, s):
        """
        Formats the string as a triple-quoted string when it contains
        newlines.
        """
        if '\n' in s:
            s = repr(s)
            s = s[0] * 3 + s[1:-1] + s[-1] * 3
            s = s.replace('\\n', '\n')
            return s
        else:
            return repr(s)

    def long_item_list(self, lst):
        """
        Returns true if the list contains items that are long, and should
        be more nicely formatted.
        """
        how_many = 0
        for item in lst:
            if len(repr(item)) > 40:
                how_many += 1
                if how_many >= 3:
                    return True
        return False