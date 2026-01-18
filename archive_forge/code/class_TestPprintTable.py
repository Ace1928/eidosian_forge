from monty.pprint import draw_tree, pprint_table
class TestPprintTable:

    def test_print(self):
        table = [['one', 'two'], ['1', '2']]
        pprint_table(table)