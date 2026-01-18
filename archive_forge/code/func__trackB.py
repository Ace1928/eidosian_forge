from reportlab.lib import colors
def _trackB(self, tracks):
    try:
        track, start, end = self.featureB
        assert track in tracks
        return track
    except TypeError:
        for track in tracks:
            for feature_set in track.get_sets():
                if hasattr(feature_set, 'features'):
                    if self.featureB in feature_set.features.values():
                        return track
        return None