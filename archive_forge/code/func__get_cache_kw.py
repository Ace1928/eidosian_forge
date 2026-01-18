from mako import util
def _get_cache_kw(self, kw, context):
    defname = kw.pop('__M_defname', None)
    if not defname:
        tmpl_kw = self.template.cache_args.copy()
        tmpl_kw.update(kw)
    elif defname in self._def_regions:
        tmpl_kw = self._def_regions[defname]
    else:
        tmpl_kw = self.template.cache_args.copy()
        tmpl_kw.update(kw)
        self._def_regions[defname] = tmpl_kw
    if context and self.impl.pass_context:
        tmpl_kw = tmpl_kw.copy()
        tmpl_kw.setdefault('context', context)
    return tmpl_kw