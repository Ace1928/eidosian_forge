import unittest
from unittest.mock import MagicMock
import sys
from pathlib import Path
import uuid

# Add root of this forge to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_forge.agent_core import AgentForge, Task

class TestAgentForge(unittest.TestCase):
    def setUp(self):
        self.forge = AgentForge()

    def test_planning(self):
        mock_llm = MagicMock()
        # LLMResponse has .text attribute
        mock_response = MagicMock()
        mock_response.text = '{"tasks": [{"tool": "t1", "description": "desc1"}]}'
        mock_llm.generate.return_value = mock_response
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
        
        task = Task(description="Do thing", task_id=str(uuid.uuid4()), tool="processor", kwargs={"input_val": "data"})
        goal.add_task(task)
        
        success = self.forge.execute_task(task)
        self.assertTrue(success)
        self.assertEqual(task.status, "completed")
        self.assertEqual(task.result, "Processed data")

    def test_missing_tool(self):
        goal = self.forge.create_goal("Fail Test", plan=False)
        task = Task(description="Nonexistent", task_id=str(uuid.uuid4()), tool="ghost")
        success = self.forge.execute_task(task)
        self.assertFalse(success)
        self.assertEqual(task.status, "failed")

    def test_summary(self):
        # Disable automatic planning for simple summary test
        goal = self.forge.create_goal("Summary Test", plan=False)
        goal.add_task(Task(description="T1", task_id=str(uuid.uuid4())))
        goal.add_task(Task(description="T2", task_id=str(uuid.uuid4())))
        
        summary = self.forge.get_task_summary()
        self.assertEqual(summary["pending"], 2)

if __name__ == "__main__":
    unittest.main()
