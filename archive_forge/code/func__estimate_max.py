def _estimate_max(self, key_count):
    """Estimate the maximum amount of 'inserting stream' work.

        This is just an estimate.
        """
    return int(key_count * 10.3)