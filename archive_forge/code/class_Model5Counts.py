import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel4
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
class Model5Counts(Counts):
    """
    Data object to store counts of various parameters during training.
    Includes counts for vacancies.
    """

    def __init__(self):
        super().__init__()
        self.head_vacancy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        self.head_vacancy_for_any_dv = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.non_head_vacancy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        self.non_head_vacancy_for_any_dv = defaultdict(lambda: defaultdict(lambda: 0.0))

    def update_vacancy(self, count, alignment_info, i, trg_classes, slots):
        """
        :param count: Value to add to the vacancy counts
        :param alignment_info: Alignment under consideration
        :param i: Source word position under consideration
        :param trg_classes: Target word classes
        :param slots: Vacancy states of the slots in the target sentence.
            Output parameter that will be modified as new words are placed
            in the target sentence.
        """
        tablet = alignment_info.cepts[i]
        tablet_length = len(tablet)
        total_vacancies = slots.vacancies_at(len(slots))
        if tablet_length == 0:
            return
        j = tablet[0]
        previous_cept = alignment_info.previous_cept(j)
        previous_center = alignment_info.center_of_cept(previous_cept)
        dv = slots.vacancies_at(j) - slots.vacancies_at(previous_center)
        max_v = total_vacancies - tablet_length + 1
        trg_class = trg_classes[alignment_info.trg_sentence[j]]
        self.head_vacancy[dv][max_v][trg_class] += count
        self.head_vacancy_for_any_dv[max_v][trg_class] += count
        slots.occupy(j)
        total_vacancies -= 1
        for k in range(1, tablet_length):
            previous_position = tablet[k - 1]
            previous_vacancies = slots.vacancies_at(previous_position)
            j = tablet[k]
            dv = slots.vacancies_at(j) - previous_vacancies
            max_v = total_vacancies - tablet_length + k + 1 - previous_vacancies
            trg_class = trg_classes[alignment_info.trg_sentence[j]]
            self.non_head_vacancy[dv][max_v][trg_class] += count
            self.non_head_vacancy_for_any_dv[max_v][trg_class] += count
            slots.occupy(j)
            total_vacancies -= 1