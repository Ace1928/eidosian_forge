import unittest
from unittest.mock import MagicMock
from eidosian_forge.agent_forge import AgentForge, Task

class TestAgentForge(unittest.TestCase):
    def setUp(self):
        self.forge = AgentForge()

    def test_planning(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {
            "success": True,
            "response": '{"tasks": [{"tool": "t1", "description": "desc1"}]}'
        }
        self.forge.llm = mock_llm
        
        goal = self.forge.create_goal("Test Goal", plan=True)
        self.assertEqual(len(goal.tasks), 1)
        self.assertEqual(goal.tasks[0].tool, "t1")

    def test_task_execution(self):
        goal = self.forge.create_goal("Tool Test", plan=False)
        
        # Define a mock tool
        def my_tool(input_val):
            return f"Processed {input_val}"
        
        self.forge.register_tool("processor", my_tool, "A test tool")
        
        task = Task("Do thing", tool="processor")
        goal.add_task(task)
        
        success = self.forge.execute_task(task, input_val="data")
        self.assertTrue(success)
        self.assertEqual(task.status, "completed")
        self.assertEqual(task.result, "Processed data")

    def test_missing_tool(self):
        goal = self.forge.create_goal("Fail Test", plan=False)
        task = Task("Nonexistent", tool="ghost")
        success = self.forge.execute_task(task)
        self.assertFalse(success)
        self.assertEqual(task.status, "failed")

    def test_summary(self):
        # Disable automatic planning for simple summary test
        goal = self.forge.create_goal("Summary Test", plan=False)
        goal.add_task(Task("T1"))
        goal.add_task(Task("T2"))
        
        summary = self.forge.get_task_summary()
        self.assertEqual(summary["pending"], 2)

if __name__ == "__main__":
    unittest.main()
