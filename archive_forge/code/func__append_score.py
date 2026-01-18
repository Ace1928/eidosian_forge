from enum import Enum
def _append_score(self, score_field, score):
    """Append SCORE_FIELD and SCORE."""
    if score_field is not None:
        self.args.append('SCORE_FIELD')
        self.args.append(score_field)
    if score is not None:
        self.args.append('SCORE')
        self.args.append(score)