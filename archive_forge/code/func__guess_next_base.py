def _guess_next_base(self, base_tree_remaining):
    import traceback
    bad_iros = C3.BAD_IROS
    if self.leaf not in bad_iros:
        if bad_iros == ():
            import weakref
            bad_iros = C3.BAD_IROS = weakref.WeakKeyDictionary()
        bad_iros[self.leaf] = t = (InconsistentResolutionOrderError(self, base_tree_remaining), traceback.format_stack())
        _logger().warning('Tracking inconsistent IRO: %s', t[0])
    return C3._guess_next_base(self, base_tree_remaining)